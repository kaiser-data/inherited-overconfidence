import { Composition } from 'remotion';
import { Promo, totalFrames, FPS } from './Promo';

export const Root: React.FC = () => (
  <Composition id="Promo" component={Promo} durationInFrames={totalFrames()} fps={FPS} width={1920} height={1080} />
);
